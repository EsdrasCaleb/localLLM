package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

public class LoadProperties_instance_0_2_Test {

    @Test
    public void testInstance() {
        // Create an instance of LoadProperties
        LoadProperties instance = LoadProperties.instance();
        // Verify that the instance is not null
        assertNotNull(instance);
        // Verify that the instance is an instance of LoadProperties
        assertTrue(instance.getClass().equals(LoadProperties.class));
    }
}
