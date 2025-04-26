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

@ExtendWith(MockitoExtension.class)
public class Query_browseNodeQueryGenerator_3_1_Test {

    // Test class
    @Test
    public void testBrowseNodeQueryGenerator() {
        String type = "";
        String page = "";
        String offer = "";
        String mode = "";
        String browseNode = "";
        String expected = "";
        String actual = "";
        Query q = new Query();
        actual = q.browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        assertEquals(expected, actual);
        // log.debug("Expected = " + expected + " Actual = " + actual);
    }
}
