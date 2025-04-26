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
public class Query_sendRequest_2_0_Test {

    @Test
    public void testSendRequest() throws Exception {
        Query query = new Query();
        query.serverURL = "http://www.example.com";
        query.searchType = "search";
        query.type = "type";
        query.page = "page";
        query.offer = "offer";
        query.searchValues = new ArrayList();
        query.searchValues.add("search");
        query.searchValues.add("type");
        query.searchValues.add("page");
        query.searchValues.add("offer");
        String response = query.sendRequest("http://www.example.com?search=search&type=type&page=page&offer=offer");
        assertEquals(response, "response");
    }
}
