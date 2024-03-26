import {
  mainloop,
  WindowCanvas,
} from "https://deno.land/x/dwm@0.3.4/ext/canvas.ts";

const canvas = new WindowCanvas({
  title: "Skia Canvas",
  width: 800,
  height: 600,
  resizable: true,
});

canvas.onDraw = (ctx) => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = "white";
  ctx.font = "30px Arial";
  ctx.textBaseline = "top";
  ctx.fillText("Hello World", 10, 10);
};

let frame = 0;
await mainloop(() => {
  console.log(frame++);
  canvas.draw();
});
