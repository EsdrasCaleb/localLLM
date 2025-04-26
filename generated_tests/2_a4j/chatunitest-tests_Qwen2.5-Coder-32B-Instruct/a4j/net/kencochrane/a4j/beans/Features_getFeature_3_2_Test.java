package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class Features_getFeature_3_2_Test {

    @InjectMocks
    private Features featuresInstance;

    @Mock
    private ArrayList<String> mockFeatures;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Injecting mockFeatures into the features field of Features class using reflection
        Field featuresField = Features.class.getDeclaredField("features");
        featuresField.setAccessible(true);
        featuresField.set(featuresInstance, mockFeatures);
    }

    @Test
    public void testGetFeature_IndexNegative() {
        // Arrange
        when(mockFeatures.size()).thenReturn(3);
        // Act
        String result = featuresInstance.getFeature(-1);
        // Assert
        assertNull(result);
        verify(mockFeatures, never()).get(-1);
    }
}
