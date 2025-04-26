package net.kencochrane.a4j.data;

import static org.junit.Assert.assertEquals;
import org.junit.Test;
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

public class Query_SearchQueryGenerator_6_0_Test {

    @Test
    public void testSearchQueryGenerator() {
        Query q = new Query();
        String expected = "http://www.example.com/search?t=0000000000&dev-t=DSB0XDDW1GQ3S&job=Java%20developer&mode=all&type=all&page=1&offer=all&f=xml";
        String actual = q.SearchQueryGenerator("job", "Java developer", "all", "all", "1", "all");
        assertEquals(expected, actual);
    }
}
