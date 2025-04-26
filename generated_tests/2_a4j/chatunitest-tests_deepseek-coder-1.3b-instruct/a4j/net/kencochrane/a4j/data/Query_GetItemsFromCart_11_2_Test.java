package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
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

public class Query_GetItemsFromCart_11_2_Test {

    @Test
    public void testGetItemsFromCart() throws Exception {
        Query query = new Query();
        Field field = Query.class.getDeclaredField("serverURL");
        field.setAccessible(true);
        field.set(query, "http://test.com");
        Field field2 = Query.class.getDeclaredField("associatesID");
        field2.setAccessible(true);
        field2.set(query, "12345");
        Field field3 = Query.class.getDeclaredField("token");
        field3.setAccessible(true);
        field3.set(query, "testToken");
        Field field4 = Query.class.getDeclaredField("searchType");
        field4.setAccessible(true);
        field4.set(query, "testType");
        Field field5 = Query.class.getDeclaredField("type");
        field5.setAccessible(true);
        field5.set(query, "testType");
        Field field6 = Query.class.getDeclaredField("page");
        field6.setAccessible(true);
        field6.set(query, "1");
        Field field7 = Query.class.getDeclaredField("offer");
        field7.setAccessible(true);
        field7.set(query, "testOffer");
        Field field8 = Query.class.getDeclaredField("searchValues");
        field8.setAccessible(true);
        field8.set(query, new ArrayList());
        String result = query.GetItemsFromCart("testCartId", "testHmac");
        assertEquals("http://test.com?ShoppingCart=get&f=xml&dev-t=testToken&t=12345&CartId=testCartId&Hmac=testHmac", result);
    }
}
