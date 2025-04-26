package net.kencochrane.a4j.data;

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
import java.util.ArrayList;
import java.util.Properties;

class Query_RemoveFromCart_13_1_Test {

    @Test
    public void testRemoveFromCart() {
        // Arrange
        Query query = new Query();
        String itemId = "17120277375791359165";
        String cartId = "CART";
        String hmac = "HMAC=";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&f=xml&type=lite";
        // Act
        String actualUrl = query.RemoveFromCart(itemId, cartId, hmac);
        // Assert
        assertEquals(expectedUrl, actualUrl);
    }
}
