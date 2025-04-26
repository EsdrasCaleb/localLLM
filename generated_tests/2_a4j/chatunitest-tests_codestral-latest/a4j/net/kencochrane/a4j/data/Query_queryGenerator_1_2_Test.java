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

public class Query_queryGenerator_1_2_Test {

    @InjectMocks
    private Query query;

    @Mock
    private a4jUtil jawsUtil;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field tokenField = Query.class.getDeclaredField("token");
        tokenField.setAccessible(true);
        tokenField.set(query, "DSB0XDDW1GQ3S");
    }

    @Test
    public void testQueryGenerator() throws Exception {
        // Arrange
        String searchType = "search";
        String type = "type";
        String page = "1";
        String offer = "offer";
        ArrayList<String> searchValues = new ArrayList<>();
        searchValues.add("value1");
        searchValues.add("value2");
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        // Act
        String result = query.queryGenerator(searchType, type, page, offer, searchValues);
        // Assert
        String expected = "http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&search=value1,value2&type=type&offerpage=1&offer=offer&f=xml";
        assertEquals(expected, result);
    }
}
