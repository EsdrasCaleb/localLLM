package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ModeList_addMode_1_0_Test {

    @InjectMocks
    private ModeList modeList;

    @Mock
    private Mode mode;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAddMode() {
        // Arrange
        ArrayList<Mode> modes = new ArrayList<>();
        modes.add(mode);
        // Act
        modeList.addMode(mode);
        // Assert
        assertEquals(modes, modeList.getAllModes());
    }
}
