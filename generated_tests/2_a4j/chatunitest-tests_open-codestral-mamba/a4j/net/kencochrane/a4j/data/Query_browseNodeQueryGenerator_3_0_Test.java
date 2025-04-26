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

public class Query_browseNodeQueryGenerator_3_0_Test {

    @Test
    public void testBrowseNodeQueryGenerator() {
        Query query = Mockito.spy(new Query());
        String type = "testType";
        String page = "testPage";
        String offer = "testOffer";
        String mode = "testMode";
        String browseNode = "testBrowseNode";
        String expectedUrl = "testUrl";
        Mockito.doReturn(expectedUrl).when(query).browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        String actualUrl = query.browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        assertEquals(expectedUrl, actualUrl);
    }
}
