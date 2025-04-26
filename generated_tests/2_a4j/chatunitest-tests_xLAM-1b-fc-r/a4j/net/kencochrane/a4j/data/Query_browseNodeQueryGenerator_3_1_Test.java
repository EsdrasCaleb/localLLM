package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
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

public class Query_browseNodeQueryGenerator_3_1_Test {

    @Test
    public void browseNodeQueryGeneratorTest() throws Exception {
        Query query = new Query();
        Field field = query.getClass().getDeclaredField("serverURL");
        field.setAccessible(true);
        field.set(query, "http://example.com");
        Method method = query.getClass().getDeclaredMethod("browseNodeQueryGenerator", String.class, String.class, String.class, String.class, String.class);
        method.setAccessible(true);
        String result = (String) method.invoke(query, "search", "1", "2", "3", "4");
        assertEquals("http://example.com?t=DSB0XDDW1GQ3S&dev-t=DSB0XDDW1GQ3S&BrowseNodeSearch=4&mode=3&type=2&page=1&offer=2&f=xml", result);
    }
}
