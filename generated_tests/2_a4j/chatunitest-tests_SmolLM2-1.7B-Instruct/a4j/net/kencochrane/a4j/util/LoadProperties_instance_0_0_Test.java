package net.kencochrane.a4j.util;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.io.File;
import java.io.IOException;
import java.util.Properties;
import static org.junit.Assert.assertNotNull;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.InputStream;

@RunWith(MockitoJUnitRunner.class)
public class LoadProperties_instance_0_0_Test {

    @Mock
    private Properties props;

    @InjectMocks
    private LoadProperties loadProperties;

    @Test
    public void testInstance() {
        when(loadProperties.getProperties()).thenReturn(props);
        assertNotNull(loadProperties.instance());
    }
}
