# Use Cases - Plant Growth Analysis Application

This document outlines all the use cases for the Plant Growth Analysis Application, organized by user roles and functional areas.

## User Roles

### Primary Users
- **Researchers**: Academic researchers studying plant growth patterns
- **Students**: Graduate and undergraduate students conducting plant research
- **Plant Enthusiasts**: Hobbyists tracking plant development
- **Agricultural Scientists**: Professionals analyzing crop growth

### Secondary Users
- **Data Analysts**: Processing exported data for statistical analysis
- **Lab Technicians**: Operating the application for data collection

## Core Use Cases

### 1. Photo Management

#### UC001: Upload Single Photo
**Actor**: User  
**Goal**: Add a single photo to the system for analysis  
**Preconditions**: User has a plant photo file on their device  
**Main Flow**:
1. User navigates to upload section
2. User clicks "Upload Photos" button or drags photo to upload area
3. User selects photo file from file browser
4. System validates image format (JPG, PNG, GIF, WebP, TIFF, BMP)
5. System extracts EXIF metadata for timestamp
6. System creates photo record with default parameters
7. System displays photo in gallery
8. System optionally processes photo with default settings

**Alternative Flows**:
- 4a. Invalid format: System displays error message
- 8a. User chooses not to process immediately

**Postconditions**: Photo is stored in database and available for processing

#### UC002: Upload Multiple Photos (Batch Upload)
**Actor**: User  
**Goal**: Add multiple photos simultaneously  
**Preconditions**: User has multiple plant photo files  
**Main Flow**:
1. User selects multiple files in upload dialog
2. System validates each file format
3. System displays progress indicator
4. For each valid photo:
   - System extracts metadata
   - System creates photo record
   - System updates progress counter
5. System displays all uploaded photos
6. User can optionally assign to collection

**Alternative Flows**:
- 2a. Some files invalid: System skips invalid files and reports errors
- 4a. Upload interrupted: System saves successfully uploaded photos

#### UC003: Assign Photo to Collection
**Actor**: User  
**Goal**: Organize photo by adding it to a collection  
**Preconditions**: Photo exists in system, collection exists or can be created  
**Main Flow**:
1. User selects photo from gallery
2. User clicks checkbox or assignment button
3. System displays collection selector dialog
4. User selects existing collection or creates new one
5. System updates photo's collection assignment
6. System updates collection's photo list
7. System confirms assignment

**Alternative Flows**:
- 4a. Create new collection: User enters collection name
- 4b. Cancel assignment: User closes dialog without selection

#### UC004: Remove Photo from Collection
**Actor**: User  
**Goal**: Unassign photo from its current collection  
**Preconditions**: Photo is assigned to a collection  
**Main Flow**:
1. User locates photo in collection or gallery
2. User unchecks assignment or clicks remove button
3. System displays confirmation dialog
4. User confirms removal
5. System removes photo from collection's photo list
6. System sets photo's collection ID to null
7. Photo remains in system but unassigned

**Alternative Flows**:
- 4a. User cancels: No changes made

#### UC005: Delete Photo
**Actor**: User  
**Goal**: Permanently remove photo from system  
**Preconditions**: Photo exists in system  
**Main Flow**:
1. User selects photo
2. User clicks delete button
3. System displays confirmation dialog with warning
4. User confirms deletion
5. System removes photo from any collections
6. System deletes photo record from database
7. System updates display to remove photo

**Alternative Flows**:
- 4a. User cancels: No deletion occurs

#### UC006: View Photo Details
**Actor**: User  
**Goal**: Examine photo information and processing results  
**Preconditions**: Photo exists in system  
**Main Flow**:
1. User clicks on photo thumbnail
2. System displays photo detail view
3. User sees original image
4. If processed, user sees processed image
5. User sees metadata (upload date, processing parameters)
6. If processed, user sees measurement results
7. User can navigate to edit mode

### 2. Collection Management

#### UC007: Create New Collection
**Actor**: User  
**Goal**: Create a named group for organizing related photos  
**Main Flow**:
1. User clicks "Create Collection" button
2. System displays collection creation dialog
3. User enters collection name
4. User optionally enters description
5. System validates name is not empty
6. System creates collection record
7. System displays new collection in list

**Alternative Flows**:
- 5a. Empty name: System shows validation error
- 3a. User cancels: No collection created

#### UC008: Rename Collection
**Actor**: User  
**Goal**: Change the name of an existing collection  
**Preconditions**: Collection exists  
**Main Flow**:
1. User selects collection
2. User clicks rename/edit button
3. System displays edit dialog with current name
4. User modifies name
5. System validates new name
6. System updates collection record
7. System refreshes display

#### UC009: Delete Collection
**Actor**: User  
**Goal**: Remove collection while preserving photos  
**Preconditions**: Collection exists  
**Main Flow**:
1. User selects collection
2. User clicks delete button
3. System displays warning that photos will remain but be unassigned
4. User confirms deletion
5. System removes collection record
6. System sets all collection photos to unassigned
7. System removes collection from display

**Alternative Flows**:
- 4a. User cancels: No deletion occurs

#### UC010: View Collection Photos
**Actor**: User  
**Goal**: Browse all photos within a specific collection  
**Preconditions**: Collection exists and contains photos  
**Main Flow**:
1. User clicks on collection
2. System navigates to collection detail view
3. System displays thumbnail grid of all collection photos
4. User sees photo count and collection metadata
5. User can select individual photos for detailed view
6. User can access bulk operations

### 3. Image Processing

#### UC011: Process Single Photo
**Actor**: User  
**Goal**: Apply computer vision analysis to extract plant measurements  
**Preconditions**: Photo is uploaded to system  
**Main Flow**:
1. User selects photo for processing
2. User opens photo editor
3. System displays original photo and empty processed area
4. User adjusts processing parameters:
   - Granularity (1-100)
   - Threshold (0-1)
   - Creation timestamp
5. User clicks "Process" button
6. System validates parameters
7. System calls Python processing script with photo data
8. Python script applies computer vision pipeline
9. System receives processed image and measurements
10. System displays side-by-side comparison
11. System saves results to database

**Alternative Flows**:
- 7a. Processing fails: System displays error message
- 8a. Invalid parameters: System shows validation errors

#### UC012: Batch Process Photos
**Actor**: User  
**Goal**: Process multiple photos with consistent parameters  
**Preconditions**: Multiple photos are available for processing  
**Main Flow**:
1. User selects multiple photos or opens collection editor
2. User enables "Process All" mode
3. User sets global parameters for all photos
4. User clicks "Process All"
5. System processes each photo sequentially with same parameters
6. System displays progress indicator
7. System updates each photo as processing completes
8. User sees all processed results

**Alternative Flows**:
- 5a. Processing error on some photos: System continues with remaining photos
- 7a. User cancels: System completes current photo then stops

#### UC013: Undo/Redo Processing Operations
**Actor**: User  
**Goal**: Revert or restore processing changes  
**Preconditions**: Photo has been processed at least once  
**Main Flow**:
1. User clicks "Undo" button
2. System restores previous photo state from backup
3. System updates display to show reverted state
4. User can click "Redo" to restore changes
5. System maintains operation history stack

**Alternative Flows**:
- 1a. No previous state: Undo button disabled
- 4a. No redo available: Redo button disabled

### 4. Data Analysis and Visualization

#### UC014: View Growth Charts
**Actor**: User  
**Goal**: Analyze plant growth trends over time  
**Preconditions**: Collection has multiple processed photos  
**Main Flow**:
1. User navigates to collection analysis view
2. System calculates growth metrics from all processed photos
3. System displays interactive charts for:
   - Height over time
   - Width over time  
   - Area over time
4. User can toggle between real-time and equal-spacing modes
5. User can interact with chart data points
6. System shows trend lines and patterns

**Alternative Flows**:
- 2a. Insufficient data: System shows message about needing more processed photos

---

## Use Case Dependencies

```
UC001 (Upload Photo) → UC011 (Process Single Photo) → UC014 (View Charts)
UC007 (Create Collection) → UC003 (Assign Photos) → UC010 (View Collection)
UC003 (Assign Photo to Collection) → UC007 (Create Collection)
UC006 (View Photo Details) → UC011 (Process Single Photo)
```

## Success Metrics

- **Usability**: Users can complete basic workflow (upload → process → analyze) within 10 minutes
- **Accuracy**: Processing algorithm correctly identifies plant boundaries in 85%+ of clear photos
- **Performance**: Processing completes within 30 seconds for typical plant photos
- **Reliability**: Application handles 100+ photo collections without performance degradation
- **Research Value**: Exported data suitable for peer-reviewed scientific analysis
