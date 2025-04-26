package net.kencochrane.a4j.data;

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

public class Query_queryGenerator_1_2_Test {

    @Mock
    private Query query;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
        query.queryGenerator("searchType", "type", "page", "offer", null);
    }

    @Test
    public void testQueryGenerator() {
        String expected = "serverURL?t=associatesID&dev-t=token&searchType=searchType&type=type&offerpage=page&offer=offer&f=xml";
        assertEquals(expected, query.queryGenerator("searchType", "type", "page", "offer", null));
    }
}
