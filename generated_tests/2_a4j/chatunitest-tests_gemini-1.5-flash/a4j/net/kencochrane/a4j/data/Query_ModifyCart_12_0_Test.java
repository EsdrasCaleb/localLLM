package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;
import net.kencochrane.a4j.util.a4jUtil;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class Query_ModifyCart_12_0_Test {

    @Mock
    private a4jUtil mockUtil;

    @Test
    void testModifyCart_normalCase() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://example.com");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "testAssociateID");
            when(mockUtil.encodeString("testHmac")).thenReturn("encodedTestHmac");
            Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
            jawsUtilField.setAccessible(true);
            jawsUtilField.set(query, mockUtil);
            String result = query.ModifyCart("123", "2", "cart123", "testHmac");
            assertEquals("http://example.com?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=testAssociateID&Item.123=2&CartId=cart123&Hmac=encodedTestHmac", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
    }

    @Test
    void testModifyCart_emptyValues() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://example.com");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "");
            when(mockUtil.encodeString("")).thenReturn("");
            Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
            jawsUtilField.setAccessible(true);
            jawsUtilField.set(query, mockUtil);
            String result = query.ModifyCart("", "", "", "");
            assertEquals("http://example.com?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=&Item.==&CartId=&Hmac=", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
    }

    @Test
    void testModifyCart_nullValues() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://example.com");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, null);
            when(mockUtil.encodeString(null)).thenReturn(null);
            Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
            jawsUtilField.setAccessible(true);
            jawsUtilField.set(query, mockUtil);
            String result = query.ModifyCart(null, null, null, null);
            assertEquals("http://example.com?ShoppingCart=modify&f=xml&dev-t=DSB0XDDW1GQ3S&t=&Item.==&CartId=&Hmac=", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
    }
}
