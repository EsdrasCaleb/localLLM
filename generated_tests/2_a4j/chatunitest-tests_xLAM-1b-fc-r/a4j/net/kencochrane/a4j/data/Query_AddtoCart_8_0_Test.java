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
    public void testAddtoCart() throws Exception {
        Query query = new Query();
        String ASIN = "1234567890";
        String quantity = "1";
        Field field = Query.class.getDeclaredField("serverURL");
        field.setAccessible(true);
        StringBuffer buffer = new StringBuffer();
        buffer.append("http://www.example.com");
        buffer.append("?");
        buffer.append("ShoppingCart=add&f=xml&dev-t=");
        buffer.append(query.token);
        buffer.append("&t=");
        buffer.append(query.associatesID);
        buffer.append("&Asin.");
        buffer.append(ASIN);
        buffer.append("=");
        buffer.append(quantity);
        String expected = buffer.toString();
        String result = query.AddtoCart(ASIN, quantity);
        assertEquals(expected, result);
        verify(query).AddtoCart(ASIN, quantity);
    }
}
