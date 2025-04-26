package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;

public class ShoppingCart_toString_8_0_Test {

    private ShoppingCart shoppingCart;

    private Items items;

    private Item item;

    @BeforeEach
    public void setUp() {
        shoppingCart = new ShoppingCart();
        items = mock(Items.class);
        item = mock(Item.class);
    }

    @Test
    public void testToString() {
        // Arrange
        shoppingCart.setHMAC("testHMAC");
        shoppingCart.setPurchaseURL("http://testpurchaseurl.com");
        shoppingCart.setCartId("testCartId");
        // Prepare mocked Item
        when(item.getQuantity()).thenReturn("2");
        when(item.getOurPrice()).thenReturn("10.00");
        // Prepare mocked Items
        ArrayList<Item> itemList = new ArrayList<>();
        itemList.add(item);
        when(items.getItemsArrayList()).thenReturn(itemList);
        shoppingCart.setItems(items);
        // Act
        String result = shoppingCart.toString();
        // Assert
        // Note: items.toString() will be called here
        String // Note: items.toString() will be called here
        expected = "HMAC = testHMAC\n" + "Purchase URL = http://testpurchaseurl.com\n" + "CartId = testCartId\n" + "items = " + items + "\n";
        assertEquals(expected, result);
    }
}
