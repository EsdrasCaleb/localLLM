package net.kencochrane.a4j.data;

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
public class Query_BlendedSearchGenerator_4_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        query.serverURL = "http://example.com";
        query.associatesID = "testAssociateID";
        query.searchType = "testSearchType";
        query.type = "testType";
        query.page = "1";
        query.offer = "testOffer";
        query.searchValues = new ArrayList<>();
    }

    @Test
    public void testBlendedSearchGenerator() {
        String type = "book";
        String searchTerm = "java";
        String encodedSearchTerm = "encodedJava";
        when(jawsUtil.encodeString(searchTerm)).thenReturn(encodedSearchTerm);
        String expectedURL = "http://example.com?t=testAssociateID&dev-t=DSB0XDDW1GQ3S&BlendedSearch=encodedJava&type=book&f=xml";
        String actualURL = query.BlendedSearchGenerator(type, searchTerm);
        assertEquals(expectedURL, actualURL);
        verify(jawsUtil, times(1)).encodeString(searchTerm);
    }
}
