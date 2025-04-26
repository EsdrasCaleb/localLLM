package org.templateit;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Iterator;
import org.apache.log4j.Logger;
import org.templateit.util.DelimitedFileReader;
import com.lowagie.text.DocumentException;

public class TemplateIt_main_0_1_Test {

    private TemplateIt templateIt;

    private String[] args;

    @BeforeEach
    public void setUp() {
        args = new String[0];
        templateIt = new TemplateIt();
    }
}
