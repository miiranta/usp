import { createRouter, createWebHistory } from 'vue-router'

import WelcomePage from '../pages/WelcomePage.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: WelcomePage,
    }
  ],
})

export default router
