package br.com.jnfe.base.util;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.KeyStore;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;

class SecurityUtils_main_8_3_Test {

    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();

    private final PrintStream originalOut = System.out;

    @BeforeEach
    public void setUpStreams() {
        System.setOut(new PrintStream(outContent));
    }

    @AfterEach
    public void restoreStreams() {
        System.setOut(originalOut);
    }

    @Test
    void testMainWithAllArguments() throws Exception {
        String[] args = { "certificateName", "certificateLocation", "trustStorePath" };
        try (MockedStatic<SecurityUtils> mockedStatic = Mockito.mockStatic(SecurityUtils.class)) {
            SecurityUtils.main(args);
            mockedStatic.verify(() -> SecurityUtils.installCertificate("trustStorePath", "certificateLocation", "certificateName"));
        }
    }

    @Test
    void testMainWithTwoArguments() throws Exception {
        String[] args = { "certificateName", "certificateLocation" };
        try (MockedStatic<SecurityUtils> mockedStatic = Mockito.mockStatic(SecurityUtils.class)) {
            SecurityUtils.main(args);
            mockedStatic.verify(() -> SecurityUtils.installCertificate(null, "certificateLocation", "certificateName"));
        }
    }

    @Test
    void testMainWithNoArguments() throws Exception {
        String[] args = {};
        SecurityUtils.main(args);
        String expectedOutput = "Uso: java SecurityUtils <localDoCertificado> <nomeDoCertificado> [localDoCacertsAPartirDoJavaHome]\n";
        assertEquals(expectedOutput, outContent.toString());
    }

    @Test
    void testMainWithOneArgument() throws Exception {
        String[] args = { "certificateName" };
        SecurityUtils.main(args);
        String expectedOutput = "Uso: java SecurityUtils <localDoCertificado> <nomeDoCertificado> [localDoCacertsAPartirDoJavaHome]\n";
        assertEquals(expectedOutput, outContent.toString());
    }
}
