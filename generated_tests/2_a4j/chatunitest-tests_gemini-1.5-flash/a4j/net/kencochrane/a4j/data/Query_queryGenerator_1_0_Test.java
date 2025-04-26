package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
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

public class Query_queryGenerator_1_0_Test {

    private Query query;

    @BeforeEach
    void setUp() {
        query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://example.com/search");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "12345");
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set up test: " + e.getMessage());
        }
    }

    @Test
    void testQueryGenerator_singleSearchValue() {
        ArrayList<String> searchValues = new ArrayList<>(Arrays.asList("java"));
        String result = query.queryGenerator("keyword", "product", "2", "offer456", searchValues);
        assertEquals("http://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&keyword=java&type=product&offerpage=2&offer=offer456&f=xml", result);
    }
}
