package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

public class LoadProperties_instance_0_0_Test {

    @Test
    public void testInstance() {
        // Test that instance() returns the same instance on subsequent calls
        LoadProperties instance1 = LoadProperties.instance();
        LoadProperties instance2 = LoadProperties.instance();
        assertEquals(instance1, instance2);
        // Test that instance() creates a new instance when it's the first call
        LoadProperties instance3 = LoadProperties.instance();
        assertNotSame(instance1, instance3);
        // Test that instance() throws an exception when the instance is not initialized
        try {
            LoadProperties instance4 = LoadProperties.instance();
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            // expected
        }
    }
}
