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

public class Query_BlendedSearchGenerator_4_1_Test {

    @Test
    public void testBlendedSearchGenerator() throws Exception {
        Query query = new Query();
        Field field = query.getClass().getDeclaredField("serverURL");
        field.setAccessible(true);
        field.set(query, "http://example.com");
        Field field2 = query.getClass().getDeclaredField("associatesID");
        field2.setAccessible(true);
        field2.set(query, "123456");
        Field field3 = query.getClass().getDeclaredField("token");
        field3.setAccessible(true);
        field3.set(query, "ABCDEF");
        Field field4 = query.getClass().getDeclaredField("searchType");
        field4.setAccessible(true);
        field4.set(query, "product");
        Field field5 = query.getClass().getDeclaredField("type");
        field5.setAccessible(true);
        field5.set(query, "category");
        Field field6 = query.getClass().getDeclaredField("page");
        field6.setAccessible(true);
        field6.set(query, "1");
        Field field7 = query.getClass().getDeclaredField("offer");
        field7.setAccessible(true);
        field7.set(query, "2");
        Field field8 = query.getClass().getDeclaredField("searchValues");
        field8.setAccessible(true);
        field8.set(query, null);
        String result = query.BlendedSearchGenerator("category", "example");
        assertEquals("http://example.com?t=123456&dev-t=ABCDEF&BlendedSearch=example&type=category&f=xml", result);
    }
}
