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

class Query_browseNodeQueryGenerator_3_4_Test {

    @Test
    void browseNodeQueryGeneratorTest() {
        // Arrange
        Query query = new Query();
        String type = "type";
        String page = "page";
        String offer = "offer";
        String mode = "mode";
        String browseNode = "browseNode";
        String expected = "serverURL?t=associatesID&dev-t=token&BrowseNodeSearch=browseNode&mode=mode&type=type&page=page&offer=offer&f=xml";
        // Act
        String result = query.browseNodeQueryGenerator(type, page, offer, mode, browseNode);
        // Assert
        assertEquals(expected, result);
    }
}
