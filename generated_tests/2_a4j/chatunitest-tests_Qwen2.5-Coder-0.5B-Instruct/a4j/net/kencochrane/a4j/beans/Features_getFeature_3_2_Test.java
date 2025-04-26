package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class Features_getFeature_3_2_Test {

    @Mock
    private ArrayList<String> features;

    @Test
    public void testGetFeature() {
        MockitoAnnotations.openMocks(this);
        when(features.size()).thenReturn(3);
        when(features.get(0)).thenReturn("Feature 1");
        when(features.get(1)).thenReturn("Feature 2");
        when(features.get(2)).thenReturn("Feature 3");
        Features featuresInstance = new Features();
        featuresInstance.setFeature(new String[] { "Feature 1", "Feature 2", "Feature 3" });
        assertEquals("Feature 1", featuresInstance.getFeature(0));
        assertEquals("Feature 2", featuresInstance.getFeature(1));
        assertEquals("Feature 3", featuresInstance.getFeature(2));
    }
}
