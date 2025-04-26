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

public class Query_AddToExistingCart_9_1_Test {

    private Query query;

    @Test
    public void testAddToExistingCart() {
        query = new Query();
        // Arrange
        String expected = "http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=A4JTEST&Asin.ASIN=1234567890&CartId=1234567890&Hmac=HMAC";
        String ASIN = "1234567890";
        String quantity = "1";
        String cartId = "1234567890";
        String hmac = "HMAC";
        // Act
        String actual = query.AddToExistingCart(ASIN, quantity, cartId, hmac);
        // Assert
        assertEquals(expected, actual);
    }
}
