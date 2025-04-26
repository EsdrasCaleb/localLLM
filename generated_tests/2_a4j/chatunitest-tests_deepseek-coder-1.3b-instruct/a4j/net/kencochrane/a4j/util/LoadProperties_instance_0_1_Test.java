package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

class LoadProperties_instance_0_1_Test {

    @Test
    void testInstance() {
        // Create two instances of LoadProperties
        LoadProperties instance1 = LoadProperties.instance();
        LoadProperties instance2 = LoadProperties.instance();
        // Assert that both instances are the same
        assertSame(instance1, instance2);
    }
}
