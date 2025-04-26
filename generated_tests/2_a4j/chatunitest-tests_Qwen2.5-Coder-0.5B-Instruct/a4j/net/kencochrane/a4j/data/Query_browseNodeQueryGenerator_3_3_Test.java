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

class Query_browseNodeQueryGenerator_3_3_Test {

    @Test
    public void testBrowseNodeQueryGenerator() {
        // Create an instance of Query
        Query query = new Query();
        // Set values for parameters
        String type = "Product";
        String page = "1";
        String offer = "2";
        String mode = "XML";
        String browseNode = "SampleNode";
        // Call the method with the set values
        String result = query.browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        // Assert the expected result
        assertEquals("http://example.com/?t=AssociatesID&dev-t=DSB0XDDW1GQ3S&BrowseNodeSearch=SampleNode&mode=XML&type=Product&page=1&offer=2", result);
    }
}
