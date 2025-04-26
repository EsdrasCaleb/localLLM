package net.kencochrane.a4j.data;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;

class Query_RemoveFromCart_13_0_Test {

    private Query query;

    private a4jUtil mockA4jUtil;

    @BeforeEach
    void setUp() {
        mockA4jUtil = Mockito.mock(a4jUtil.class);
        query = new Query();
        query.serverURL = "http://xml.amazon.com/onca/xml3";
        query.associatesID = "test";
        // Mock the encodeString method
        Mockito.when(mockA4jUtil.encodeString(Mockito.anyString())).thenReturn("HMAC=encodedValue");
        query.jawsUtil = mockA4jUtil;
        query.searchValues = new ArrayList<>();
    }

    @Test
    void testRemoveFromCart() {
        String itemId = "123";
        String cartId = "CART";
        String hmac = "someHMAC";
        String expectedURL = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&f=xml&dev-t=DSB0XDDW1GQ3S&t=test&Item.123&CartId=CART&Hmac=HMAC=encodedValue";
        String actualURL = query.RemoveFromCart(itemId, cartId, hmac);
        assertEquals(expectedURL, actualURL);
    }

    @Test
    void testRemoveFromCart_NullItemId() {
        String itemId = null;
        String cartId = "CART";
        String hmac = "someHMAC";
        assertThrows(NullPointerException.class, () -> query.RemoveFromCart(itemId, cartId, hmac));
    }

    @Test
    void testRemoveFromCart_NullCartId() {
        String itemId = "123";
        String cartId = null;
        String hmac = "someHMAC";
        assertThrows(NullPointerException.class, () -> query.RemoveFromCart(itemId, cartId, hmac));
    }

    @Test
    void testRemoveFromCart_NullHmac() {
        String itemId = "123";
        String cartId = "CART";
        String hmac = null;
        assertThrows(NullPointerException.class, () -> query.RemoveFromCart(itemId, cartId, hmac));
    }
}
