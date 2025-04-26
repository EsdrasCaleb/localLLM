package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class LoadProperties_instance_0_3_Test {

    // JUnit class
    @Test
    public void testLoadProperties() {
        LoadProperties instance = LoadProperties.instance();
        assertNotNull(instance);
    }
}
