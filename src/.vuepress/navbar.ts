import { navbar } from "vuepress-theme-hope";

export default navbar([
  "/",
  {
    text: "文章",
    icon: "book",
    link: "/article",
  },
  "/travel",
  {
    text: "时间轴",
    icon: "timeline",
    link: "/timeline",
  },
  {
    text: "关于我",
    link: "/intro"
  },
]);
