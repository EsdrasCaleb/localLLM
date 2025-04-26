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

class Query_AddtoCart_8_0_Test {

    @Test
    public void testAddtoCart() {
        // Create an instance of the Query class
        Query query = new Query();
        // Set the expected values for the parameters
        String ASIN = "ASIN123";
        String quantity = "1";
        // Call the AddtoCart method with the expected values
        String result = query.AddtoCart(ASIN, quantity);
        // Verify that the result is as expected
        assertEquals("http://example.com/AddToCart?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=associatesID&Asin.=" + ASIN + "&quantity=" + quantity, result);
    }
}
