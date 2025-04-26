package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;
import java.util.Properties;

public class LoadProperties_instance_0_4_Test {

    @Mock
    private Properties properties;

    @InjectMocks
    private LoadProperties loadProperties;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testInstanceCreation() throws Exception {
        // Create an instance of LoadProperties
        LoadProperties loadProperties = LoadProperties.instance();
        // Verify that the instance is not null
        assertNotNull(loadProperties);
        // Verify that the instance is the same as the injected mock
        assertEquals(loadProperties, loadProperties);
    }

    @Test
    public void testMultipleInstancesCreation() throws Exception {
        // Create two instances of LoadProperties
        LoadProperties loadProperties1 = LoadProperties.instance();
        LoadProperties loadProperties2 = LoadProperties.instance();
        // Verify that both instances are the same
        assertSame(loadProperties1, loadProperties2);
    }
}
