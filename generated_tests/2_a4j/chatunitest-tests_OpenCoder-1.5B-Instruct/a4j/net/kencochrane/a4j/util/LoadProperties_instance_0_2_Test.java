package net.kencochrane.a4j.util;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

public class LoadProperties_instance_0_2_Test {

    // Create a new instance of the LoadProperties class
    LoadProperties instance = LoadProperties.instance();

    @Test
    public void testGetInstance() {
        // Create a new instance of the LoadProperties class
        LoadProperties instance = LoadProperties.instance();
        // Assert that the instance is not null
        Assertions.assertNotNull(instance);
        // Assert that the instance is the same instance
        Assertions.assertSame(instance, LoadProperties.instance());
    }
}
