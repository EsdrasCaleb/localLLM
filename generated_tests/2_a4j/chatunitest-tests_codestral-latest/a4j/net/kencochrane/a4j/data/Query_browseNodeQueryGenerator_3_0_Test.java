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

public class Query_browseNodeQueryGenerator_3_0_Test {

    @InjectMocks
    private Query query;

    @Mock
    private a4jUtil jawsUtil;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://example.com");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "testAssociatesID");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
    }

    @Test
    public void testBrowseNodeQueryGenerator() {
        String type = "Books";
        String page = "1";
        String offer = "All";
        String mode = "browse";
        String browseNode = "123456";
        String expectedQuery = "http://example.com?t=testAssociatesID&dev-t=DSB0XDDW1GQ3S&BrowseNodeSearch=123456&mode=browse&type=Books&page=1&offer=All&f=xml";
        String actualQuery = query.browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        assertEquals(expectedQuery, actualQuery);
    }

    @Test
    public void testBrowseNodeQueryGeneratorWithNullValues() {
        String type = null;
        String page = null;
        String offer = null;
        String mode = null;
        String browseNode = null;
        String expectedQuery = "http://example.com?t=testAssociatesID&dev-t=DSB0XDDW1GQ3S&BrowseNodeSearch=null&mode=null&type=null&page=null&offer=null&f=xml";
        String actualQuery = query.browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        assertEquals(expectedQuery, actualQuery);
    }

    @Test
    public void testBrowseNodeQueryGeneratorWithEmptyValues() {
        String type = "";
        String page = "";
        String offer = "";
        String mode = "";
        String browseNode = "";
        String expectedQuery = "http://example.com?t=testAssociatesID&dev-t=DSB0XDDW1GQ3S&BrowseNodeSearch=&mode=&type=&page=&offer=&f=xml";
        String actualQuery = query.browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        assertEquals(expectedQuery, actualQuery);
    }
}
