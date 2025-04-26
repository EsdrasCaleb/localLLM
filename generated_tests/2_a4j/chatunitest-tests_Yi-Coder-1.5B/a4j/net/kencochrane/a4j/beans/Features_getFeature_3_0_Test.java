package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class Features_getFeature_3_0_Test {

    @Test
    public void testGetFeature() {
        Features features = new Features();
        features.setFeature(new String[] { "Feature 1", "Feature 2" });
        assertEquals("Feature 1", features.getFeature(0));
        assertEquals("Feature 2", features.getFeature(1));
    }
}
