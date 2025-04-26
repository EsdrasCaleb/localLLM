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

    @InjectMocks
    private ShoppingCart shoppingCart;

    @Mock
    private Items items;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString() {
        // Set up the shopping cart
        shoppingCart.setHMAC("testHMAC");
        shoppingCart.setPurchaseURL("testPurchaseURL");
        shoppingCart.setCartId("testCartId");
        shoppingCart.setItems(items);
        // Define the expected string
        String expected = "HMAC = testHMAC\n" + "Purchase URL = testPurchaseURL\n" + "CartId = testCartId\n" + "items = " + items.toString() + "\n";
        // Call the toString method
        String result = shoppingCart.toString();
        // Verify the result
        assertEquals(expected, result);
    }
}
