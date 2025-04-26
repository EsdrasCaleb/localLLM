package net.kencochrane.a4j.data;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
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

@RunWith(MockitoJUnitRunner.class)
public class Query_SearchThirdPartyGenerator_7_1_Test {

    @Mock
    private Query query;

    @Test
    public void testSearchThirdPartyGenerator() {
        String sellerId = "12345";
        String type = "1";
        String page = "1";
        String status = "1";
        String expected = "http://www.a4j.com/?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=12345&type=1&page=1&offerstatus=1&f=xml";
        when(query.SearchThirdPartyGenerator(sellerId, type, page, status)).thenReturn(expected);
        String actual = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        Assert.assertEquals(expected, actual);
    }
}
