package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
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

public class Query_AddtoCart_8_0_Test {

    @Test
    void testAddtoCart_validInput() throws Exception {
        Query query = new Query();
        // Using reflection to set private fields.  Normally you'd use setters if available.
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        String result = query.AddtoCart("B07XYZ1234", "2");
        assertEquals("https://example.com?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Asin.B07XYZ1234=2", result);
    }

    @Test
    void testAddtoCart_emptyASIN() throws Exception {
        Query query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        String result = query.AddtoCart("", "2");
        assertEquals("https://example.com?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Asin.=2", result);
    }

    @Test
    void testAddtoCart_emptyQuantity() throws Exception {
        Query query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        String result = query.AddtoCart("B07XYZ1234", "");
        assertEquals("https://example.com?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Asin.B07XYZ1234=", result);
    }

    @Test
    void testAddtoCart_nullASIN() throws Exception {
        Query query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        String result = query.AddtoCart(null, "2");
        assertEquals("https://example.com?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Asin.=2", result);
    }

    @Test
    void testAddtoCart_nullQuantity() throws Exception {
        Query query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        String result = query.AddtoCart("B07XYZ1234", null);
        assertEquals("https://example.com?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Asin.B07XYZ1234=", result);
    }
}
