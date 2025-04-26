// Test class
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class ModeList_addMode_1_1_Test {

    @Mock
    private ModeList modeList;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testAddMode() {
        // Arrange
        Mode mode = new Mode();
        // Act
        modeList.addMode(mode);
        // Assert
        verify(modeList).addMode(mode);
    }
}
