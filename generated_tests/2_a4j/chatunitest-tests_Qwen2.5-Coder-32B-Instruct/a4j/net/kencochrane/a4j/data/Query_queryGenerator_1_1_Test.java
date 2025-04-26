package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.LoadProperties;
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
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class Query_queryGenerator_1_1_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize protected fields using reflection
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        Field tokenField = Query.class.getDeclaredField("token");
        tokenField.setAccessible(true);
        tokenField.set(query, "DSB0XDDW1GQ3S");
        Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
        jawsUtilField.setAccessible(true);
        jawsUtilField.set(query, jawsUtil);
    }

    @Test
    public void testQueryGenerator() {
        // Arrange
        String searchType = "books";
        String type = "fiction";
        String page = "1";
        String offer = "special";
        ArrayList<String> searchValues = new ArrayList<>();
        searchValues.add("Java");
        searchValues.add("JUnit");
        String expectedSearchString = "Java+JUnit";
        when(jawsUtil.arrayToCommaString(searchValues)).thenReturn(expectedSearchString);
        String expectedQuery = "http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&books=Java+JUnit&type=fiction&offerpage=1&offer=special&f=xml";
        // Act
        String actualQuery = query.queryGenerator(searchType, type, page, offer, searchValues);
        // Assert
        assertEquals(expectedQuery, actualQuery);
        verify(jawsUtil, times(1)).arrayToCommaString(searchValues);
    }
}
