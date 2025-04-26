package net.kencochrane.a4j.data;

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

public class Query_queryGenerator_1_1_Test {

    private Query query;

    private String searchType = "searchType";

    private String type = "type";

    private String page = "page";

    private String offer = "offer";

    private ArrayList<String> searchValues = new ArrayList<String>();

    @BeforeEach
    public void setUp() {
        query = new Query();
        searchValues.add("value1");
        searchValues.add("value2");
    }

    @Test
    public void testQueryGenerator() {
        String expected = "serverURL?t=associatesID&dev-t=DSB0XDDW1GQ3S&searchType=value1,value2&type=type&offerpage=page&offer=offer&f=xml";
        assertEquals(expected, query.queryGenerator(searchType, type, page, offer, searchValues));
    }
}
