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

public class Query_ClearCart_10_1_Test {

    @Test
    void testClearCart_validInput() throws Exception {
        Query query = new Query();
        // Using reflection to set private fields for testing purposes.  Avoid this in production code if possible.
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "testAssociateID");
        a4jUtil mockUtil = Mockito.mock(a4jUtil.class);
        Mockito.when(mockUtil.encodeString("testHmac")).thenReturn("encodedTestHmac");
        Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
        jawsUtilField.setAccessible(true);
        jawsUtilField.set(query, mockUtil);
        String result = query.ClearCart("testCartId", "testHmac");
        assertEquals("http://example.com?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociateID&CartId=testCartId&Hmac=encodedTestHmac", result);
    }

    @Test
    void testClearCart_nullCartId() throws Exception {
        Query query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "testAssociateID");
        a4jUtil mockUtil = Mockito.mock(a4jUtil.class);
        Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
        jawsUtilField.setAccessible(true);
        jawsUtilField.set(query, mockUtil);
        String result = query.ClearCart(null, "testHmac");
        // Expecting null to be handled as "null" string
        assertEquals("http://example.com?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociateID&CartId=null&Hmac=", result);
    }

    @Test
    void testClearCart_nullHmac() throws Exception {
        Query query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "testAssociateID");
        a4jUtil mockUtil = Mockito.mock(a4jUtil.class);
        Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
        jawsUtilField.setAccessible(true);
        jawsUtilField.set(query, mockUtil);
        String result = query.ClearCart("testCartId", null);
        // Expecting null to be handled as empty string
        assertEquals("http://example.com?ShoppingCart=clear&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociateID&CartId=testCartId&Hmac=", result);
    }

    @Test
    void testClearCart_nullServerURL() throws Exception {
        Query query = new Query();
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, null);
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "testAssociateID");
        a4jUtil mockUtil = Mockito.mock(a4jUtil.class);
        Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
    }
}
