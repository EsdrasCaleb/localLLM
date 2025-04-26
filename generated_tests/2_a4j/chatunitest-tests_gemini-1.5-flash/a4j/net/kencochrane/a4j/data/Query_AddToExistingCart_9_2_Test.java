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

class Query_AddToExistingCart_9_2_Test {

    @Test
    void AddToExistingCart_normalCase() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "myAssociateID");
            a4jUtil mockUtil = Mockito.mock(a4jUtil.class);
            Mockito.when(mockUtil.encodeString("myHmac")).thenReturn("encodedHmac");
            Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
            jawsUtilField.setAccessible(true);
            jawsUtilField.set(query, mockUtil);
            String result = query.AddToExistingCart("B0792H6R2R", "2", "myCartId", "myHmac");
            assertEquals("http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=myAssociateID&Asin.B0792H6R2R=2&CartId=myCartId&Hmac=encodedHmac", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception during reflection: " + e.getMessage());
        }
    }

    @Test
    void AddToExistingCart_emptyInputs() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "myAssociateID");
            a4jUtil mockUtil = Mockito.mock(a4jUtil.class);
            Mockito.when(mockUtil.encodeString("")).thenReturn("");
            Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
            jawsUtilField.setAccessible(true);
            jawsUtilField.set(query, mockUtil);
            String result = query.AddToExistingCart("", "", "", "");
            assertEquals("http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=myAssociateID&Asin.==&CartId=&Hmac=", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception during reflection: " + e.getMessage());
        }
    }

    @Test
    void AddToExistingCart_nullInputs() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "myAssociateID");
            a4jUtil mockUtil = Mockito.mock(a4jUtil.class);
            Mockito.when(mockUtil.encodeString(null)).thenReturn("");
            Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
            jawsUtilField.setAccessible(true);
            jawsUtilField.set(query, mockUtil);
            String result = query.AddToExistingCart(null, null, null, null);
            assertEquals("http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=myAssociateID&Asin.==&CartId=&Hmac=", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
        }
    }
}
