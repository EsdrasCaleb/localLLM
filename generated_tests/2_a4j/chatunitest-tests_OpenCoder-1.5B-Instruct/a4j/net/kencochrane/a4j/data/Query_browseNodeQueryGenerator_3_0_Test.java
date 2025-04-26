package net.kencochrane.a4j.data;

import java.util.Arrays;
import java.util.List;
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

public class Query_browseNodeQueryGenerator_3_0_Test {

    private Query query = new Query();

    @Test
    public void testBrowseNodeQueryGenerator() {
        String expected = "http://example.com/?t=12345&dev-t=DSB0XDDW1GQ3S&BrowseNodeSearch=browseNode&mode=mode&type=type&page=page&offer=offer&f=xml";
        String actual = query.browseNodeQueryGenerator("type", "page", "offer", "mode", "browseNode");
        assertEquals(expected, actual);
    }
}
