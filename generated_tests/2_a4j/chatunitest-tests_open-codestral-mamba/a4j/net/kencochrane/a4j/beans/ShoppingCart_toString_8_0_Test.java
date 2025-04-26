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

class ShoppingCart_toString_8_0_Test {

    private ShoppingCart cart;

    @BeforeEach
    public void setup() {
        cart = new ShoppingCart();
        Items items = Mockito.mock(Items.class);
        cart.setItems(items);
    }

    @Test
    void testToString() {
        String expected = "HMAC = null\n" + "Purchase URL = null\n" + "CartId = null\n" + "items = Items{itemsArrayList=null}\n";
        assertEquals(expected, cart.toString());
    }
}
