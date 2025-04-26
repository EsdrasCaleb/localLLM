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

public class Query_AddtoCart_8_2_Test {

    @Test
    public void testAddtoCart() {
        // Arrange
        Query query = new Query();
        String ASIN = "1234567890";
        String quantity = "1";
        String expectedUrl = "http://www.example.com/?ShoppingCart=add&f=xml&dev-t=" + query.token + "&t=" + query.associatesID + "&Asin." + ASIN + "=" + quantity;
        // Act
        String result = query.AddtoCart(ASIN, quantity);
        // Assert
        assertEquals(expectedUrl, result);
    }
}
