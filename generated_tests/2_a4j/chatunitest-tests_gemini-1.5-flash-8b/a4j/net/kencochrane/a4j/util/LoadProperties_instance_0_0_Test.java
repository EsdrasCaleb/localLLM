package net.kencochrane.a4j.util;

import java.util.Properties;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;

class LoadProperties_instance_0_0_Test {

    @Test
    void testInstance() throws NoSuchFieldException, IllegalAccessException {
        // Test case 1: First call to instance()
        LoadProperties instance1 = LoadProperties.instance();
        LoadProperties instance2 = LoadProperties.instance();
        // Verify that both instances are the same object.
        Assertions.assertSame(instance1, instance2);
        // Test case 2: Verify properties are set correctly.
        Properties properties = instance1.getProperties();
        Assertions.assertEquals("popcornmonste2-20", properties.getProperty("associateID"));
        Assertions.assertEquals("86400000", properties.getProperty("cacheLife"));
        Assertions.assertEquals("/", properties.getProperty("cacheDir"));
        Assertions.assertEquals("- ", properties.getProperty("URLSeperator"));
        Assertions.assertEquals("http://xml.amazon.net/onca/xml3", properties.getProperty("amazonServerURL"));
    }
}
