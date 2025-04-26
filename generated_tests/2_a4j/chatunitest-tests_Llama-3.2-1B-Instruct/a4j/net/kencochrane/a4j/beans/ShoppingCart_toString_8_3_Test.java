package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class ShoppingCart_toString_8_3_Test {

    @Mock
    private Items items;

    @InjectMocks
    private ShoppingCart cart;

    @Test
    public void testToString() {
        // Given
        when(items.getItemsArrayList()).thenReturn(new ArrayList<>());
        // When
        String result = cart.toString();
        // Then
        // The result should include the HMAC, purchase URL, cart ID, and the contents of the cart.
        String expected = "HMAC = HMAC, Purchase URL = purchaseURL, CartId = cartId, items = [1, 2, 3, 4, 5]";
        assertEquals(expected, result);
    }
}
