// Test method
package net.kencochrane.a4j.util;

import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;

public class LoadProperties_instance_0_2_Test {

    @Test
    public void testInstanceCreation() {
        Properties props = new Properties();
        props.setProperty("associateID", "popcornmonste2-20");
        props.setProperty("cacheLife", "86400000");
        props.setProperty("URLSeperator", "-");
        props.setProperty("cacheDir", "/");
        props.setProperty("amazonServerURL", "http://xml.amazon.net/onca/xml3");
        LoadProperties instance = LoadProperties.instance();
        assertEquals("Loaded properties", props.getProperty("associateID"), instance.getProperties().getProperty("associateID"));
        assertEquals("Loaded properties", props.getProperty("cacheLife"), instance.getProperties().getProperty("cacheLife"));
        assertEquals("Loaded properties", props.getProperty("URLSeperator"), instance.getProperties().getProperty("URLSeperator"));
        assertEquals("Loaded properties", props.getProperty("cacheDir"), instance.getProperties().getProperty("cacheDir"));
        assertEquals("Loaded properties", props.getProperty("amazonServerURL"), instance.getProperties().getProperty("amazonServerURL"));
    }
}
