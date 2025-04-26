package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class Query_SearchQueryGenerator_6_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "testID");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
    }

    @Test
    public void testSearchQueryGenerator() throws Exception {
        String searchType = "keyword";
        String searchTerm = "test";
        String mode = "search";
        String type = "books";
        String page = "1";
        String offer = "all";
        when(jawsUtil.encodeString(searchTerm)).thenReturn("encodedTest");
        String expectedQuery = "http://example.com?t=testID&dev-t=DSB0XDDW1GQ3S&keyword=encodedTest&mode=search&type=books&page=1&offer=all&f=xml";
        String actualQuery = query.SearchQueryGenerator(searchType, searchTerm, mode, type, page, offer);
        assertEquals(expectedQuery, actualQuery);
    }
}
