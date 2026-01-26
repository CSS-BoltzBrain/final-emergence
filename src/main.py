from shopmap import ShopMap
from statemap import StateMap

if __name__ == "__main__":
    state_map = StateMap("configs/empty.yaml", scale_factor=2)
    shop_map = state_map.get_shop()
    agent_map = state_map.get_agent_map()
    print(shop_map.layout_array)
    print(shop_map.products_list)
    shop_map._plot_layout(shop_map.layout_array)
