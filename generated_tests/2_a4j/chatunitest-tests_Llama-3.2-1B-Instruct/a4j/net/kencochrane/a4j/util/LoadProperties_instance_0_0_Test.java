package net.kencochrane.a4j.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.InputStream;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class LoadProperties_instance_0_0_Test {

    @Mock
    private LoadProperties _instance;

    @InjectMocks
    private LoadProperties instance;

    @Test
    public void testInstance() {
        // Arrange
        // No inputs
        // Does nothing (just creates a new instance)
        // Returns a new instance of LoadProperties
        // Act
        LoadProperties result = instance.instance();
        // Assert
        assertNotNull(result);
    }
}
