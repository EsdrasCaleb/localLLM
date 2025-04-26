package net.kencochrane.a4j.util;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

public class LoadProperties_instance_0_0_Test {

    @BeforeEach
    public void setUp() throws Exception {
        // Reset the singleton instance before each test
        Field instance = LoadProperties.class.getDeclaredField("_instance");
        instance.setAccessible(true);
        instance.set(null, null);
    }

    @AfterEach
    public void tearDown() throws Exception {
        // Reset the singleton instance after each test
        Field instance = LoadProperties.class.getDeclaredField("_instance");
        instance.setAccessible(true);
        instance.set(null, null);
    }

    @Test
    public void testInstance() {
        // Test that the instance method returns the same instance
        LoadProperties instance1 = LoadProperties.instance();
        LoadProperties instance2 = LoadProperties.instance();
        assertSame(instance1, instance2, "The instance method should return the same instance");
    }

    @Test
    public void testInstanceNotNull() {
        // Test that the instance method does not return null
        LoadProperties instance = LoadProperties.instance();
        assertNotNull(instance, "The instance method should not return null");
    }
}
